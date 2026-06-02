(define (problem picture_17)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_regulator green_regulator red_sensor - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear red_regulator)
        (clear green_regulator)
        (clear red_sensor)
        (part_at red_regulator table)
        (part_at green_regulator table)
        (part_at red_sensor table)
    )
    
    (:goal
        (and
            (part_at red_regulator regulator_placement)
        )
    )
)
