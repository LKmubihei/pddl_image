(define (problem picture_45)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_regulator red_sensor red_pump - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear red_regulator)
        (clear red_sensor)
        (clear red_pump)
        (part_at red_regulator table)
        (part_at red_sensor table)
        (part_at red_pump table)
    )
    
    (:goal
        (and
            (part_at red_pump pump_placement)
            (part_at red_regulator regulator_placement)
        )
    )
)
