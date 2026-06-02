(define (problem picture_44)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_regulator blue_pump red_pump - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear red_regulator)
        (clear blue_pump)
        (clear red_pump)
        (part_at red_regulator table)
        (part_at blue_pump table)
        (part_at red_pump table)
    )
    
    (:goal
        (and
            (part_at blue_pump pump_placement)
            (part_at red_regulator regulator_placement)
        )
    )
)
